# MPK Execution Flow Short Notes

Here I have traced one operator, RMSNorm, through the full path MPK takes to run it: the Python description, the C++ registration step that turns it into CUDA source, the pass that glues every operator's generated source into a single dispatcher, and the persistent kernel that executes it.

## Python Layer

Every operator in the model is declared through a small Python helper. RMSNorm's looks like this (`persistent_kernel.py:650`):

```python
def rmsnorm_layer(self, input, weight, output, grid_dim, block_dim):
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(input,  (0, -1, -1), 1, True)
    tb_graph.new_input(weight, (-1, -1, -1), 0, True)
    tb_graph.new_input(output, (0, -1, -1), 1, True)
    self.kn_graph.customized([input, weight, output], tb_graph)
    self.kn_graph.register_task(tb_graph, "rmsnorm") 
    # Originally register_task(tb_graph, "rmsnorm_hopper" if self.target_cc >= 90 else "rmsnorm")
```

`TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))` builds a description object. It records that this operator splits into `grid_dim` tasks, each running `block_dim` threads. Nothing executes at this point.

Each tensor is then registered through `new_input`, output included. The split between inputs and outputs is not decided here. It is resolved later, from a fixed count kept on the C++ side. The tuple following each tensor is its partition map: position in the tuple corresponds to a grid axis (x, y, z), and the value at that position names which tensor dimension is tied to that axis. A `-1` means the axis does not slice that tensor at all. `input`'s map, `(0, -1, -1)`, ties grid axis x to tensor dimension 0, the 64 tokens, so each task owns one token's row. `weight`'s map, `(-1, -1, -1)`, ties nothing, so every task receives the same full 2048-element vector.

`grid_dim` does not mean what it would in ordinary CUDA. There is exactly one kernel launch in the entire system, covered under "Persistent kernel" below. `grid_dim` here is not a launch grid. It is a task count. `grid_dim=(64,1,1)` means the operator is split into 64 tasks. `block_dim=(128,1,1)` is the thread count each of those tasks runs with once dispatched.

The final line, `register_task(tb_graph, "rmsnorm")`, is the handoff to C++. That string is the only information used to decide which CUDA gets generated for this operator.

## Task Registration

`Graph::register_task` (`graph.cc:439`) receives that string, takes the operator most recently added to the graph, and dispatches on the name:

```cpp
void Graph::register_task(char const *task_type, std::vector<int> params) {
  KNOperator const *op = operators.back();
  ...
  } else if (name == "rmsnorm") {
    int variant_id = task_register->register_rmsnorm_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_RMS_NORM, variant_id);
  }
```

The tuple recorded here, 2 inputs, 1 output, this operator's task type, and a variant id, is what later determines where the input/output boundary falls.

`register_rmsnorm_task` (`task_register.cc:91`) is where CUDA source is produced, one line at a time, through a small helper, `code.e`, that behaves like a formatted print statement, with `$` standing in for a value to be substituted:

```cpp
int batch_size = 1, hidden_dim = 2048;   // this task's slice, not the full tensor
code.e("kernel::rms_norm_impl<bfloat16, $, $>(", batch_size, hidden_dim);
code.e("    task_desc->input_ptrs[0],");
code.e("    task_desc->input_ptrs[1],");
code.e("    task_desc->output_ptrs[0],");
code.e("    1e-6f);");
```

With those two values substituted, the result is the following text:

```cpp
kernel::rms_norm_impl<bfloat16, 1, 2048>(
    task_desc->input_ptrs[0],
    task_desc->input_ptrs[1],
    task_desc->output_ptrs[0],
    1e-6f);
```

The shapes, 1 and 2048, are written directly into the generated source as template arguments. They are fixed at compile time. The tensors are not. They remain as `input_ptrs[i]` and `output_ptrs[i]`, lookups into fields on the task descriptor that are populated with real memory addresses only once the task is dispatched at runtime. Shapes are resolved statically. Addresses are resolved dynamically. That is the underlying design of the registration step.

## Gluing Into One Kernel

Once every operator in the model has gone through its own version of the step above, a second pass, `generate_task_graph` (`runtime.cc:1876`), walks the complete set of registered variants and assembles them into a single dispatcher function:

```cpp
void _execute_task(TaskDesc const* task_desc, RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_RMS_NORM && task_desc->variant_id == 0) {
    kernel::rms_norm_impl<bfloat16, 1, 2048>( ... );   // pasted in unchanged from registration
  }
  else if (task_desc->task_type == TASK_LINEAR && task_desc->variant_id == 0) { ... }
  // one branch for every (operator, shape) combination present in the model
}
```

`_execute_task` performs no computation of its own. It is an if/else chain keyed on `task_type` and `variant_id`, and its only function is routing execution to whichever generated snippet matches.

The resulting text, along with the task-graph construction code produced in the same pass, is concatenated with a second, fixed block of hand-written C called `HARD_CODE`, which provides the Python extension glue: `init_func`, `launch_func`, and related entry points. The combined source is written to a single file and handed to `nvcc`:

```python
results = self.kn_graph.generate_task_graph(...)
with open("test.cu", "w") as f:
    f.write(results["cuda_code"] + HARD_CODE)
subprocess.run(["nvcc", "test.cu", "-o", "test.so", ...])
```

The compiled `.so` is loaded back into the Python process, and calling into it triggers execution:

```
self.launch_func(...)                  # Python, into the compiled .so
  launch_func(...)                     # C, sets up CUDA streams
    launch_persistent_kernel(stream)   # C++. the one <<<...>>> launch in the system
      persistent_kernel<<<dim3(num_sms,1,1), dim3(THREADS,1,1), SHMEM>>>(config);
```

There is exactly one launch. After it, the CPU has no further role.

## Persistent Kernel

Once launched, every worker block runs the same loop for the remainder of the program:

```
while (true) {
    wait for a task id on my queue           // __nanosleep(10) between checks
    fetch its TaskDesc
    if task_type == TERMINATE: return
    else: _execute_task(task_desc, config)   // the dispatcher described above
    decrement the counter on whatever this task's completion unblocks
}
```

The mechanism behind "decrement the counter" is `EventDesc`:

```cpp
struct EventDesc {
  EventType event_type;
  int num_triggers;                     // producer tasks required before this fires
  TaskId first_task_id, last_task_id;   // task range released once it does
};
```

Each completed task atomically decrements `num_triggers` on the event it feeds. When that count reaches zero, the scheduler releases the range `[first_task_id, last_task_id)` onto the worker queues. There is no dependency graph traversal at runtime. Dependencies are expressed entirely as atomic counters reaching zero.

`TaskId` also encodes the decode iteration it belongs to, `iteration << 32 | position`. This is what allows the same task graph to be replayed for every new token without being rebuilt.
