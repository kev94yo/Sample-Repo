# ❓ Questions & Help

## Details

<!-- Description of your question -->
Hi! First, thanks for the interesting research and library :) I'm trying to run the example notebook ('example/getting-started-session-based/02-session-based-XLNet-with-PyT.ipynb'), but am running into several errors seemingly related with PyTorch, etc.

May I ask the recommended library installation process and environment settings to successfully run the example notebooks without errors? My current environment is:

conda, CUDA Version: 11.4  
python 3.7.12, nvtabular 0.7.1, transformers4rec 0.1.2, pytorch 1.7.1, cudatoolkit 11.0.221 

I got to this environment by installing in this order:
1. NVTabular  
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=11.0
2. TRM4Rec  
conda install -c nvidia -c rapidsai -c numba -c conda-forge transformers4rec cudatoolkit=11.0
3. PyTorch  
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

Thanks for the help!

Example error that I am having currently:
I first got this cuda related error by running the script than instantiates training.
```python
start_time_window_index = 1
final_time_window_index = 7
#Iterating over days of one week
for time_index in range(start_time_window_index, final_time_window_index):
    # Set data 
    time_index_train = time_index
    time_index_eval = time_index + 1
    train_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_train}/train.parquet"))
    eval_paths = glob.glob(os.path.join(OUTPUT_DIR, f"{time_index_eval}/valid.parquet"))
    print(train_paths)
    
    # Train on day related to time_index 
    print('*'*20)
    print("Launch training for day %s are:" %time_index)
    print('*'*20 + '\n')
    trainer.train_dataset_or_path = train_paths
    trainer.reset_lr_scheduler()
    trainer.train()
    trainer.state.global_step +=1
    print('finished')
    
    # Evaluate on the following day
    trainer.eval_dataset_or_path = eval_paths
    train_metrics = trainer.evaluate(metric_key_prefix='eval')
    print('*'*20)
    print("Eval results for day %s are:\t" %time_index_eval)
    print('\n' + '*'*20 + '\n')
    for key in sorted(train_metrics.keys()):
        print(" %s = %s" % (key, str(train_metrics[key]))) 
    wipe_memory()
```
```python
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_87597/2595200145.py in <module>
     16     trainer.train_dataset_or_path = train_paths
     17     trainer.reset_lr_scheduler()
---> 18     trainer.train()
     19     trainer.state.global_step +=1
     20     print('finished')

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   1288             self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
   1289 
-> 1290             for step, inputs in enumerate(epoch_iterator):
   1291 
   1292                 # Skip past any already trained steps if resuming training

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/utils/data/dataloader.py in __next__(self)
    433         if self._sampler_iter is None:
    434             self._reset()
--> 435         data = self._next_data()
    436         self._num_yielded += 1
    437         if self._dataset_kind == _DatasetKind.Iterable and \

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/utils/data/dataloader.py in _next_data(self)
    473     def _next_data(self):
    474         index = self._next_index()  # may raise StopIteration
--> 475         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    476         if self._pin_memory:
    477             data = _utils.pin_memory.pin_memory(data)

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     26             for _ in possibly_batched_index:
     27                 try:
---> 28                     data.append(next(self.dataset_iter))
     29                 except StopIteration:
     30                     break

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in __next__(self)
    332 
    333     def __next__(self):
--> 334         return self._get_next_batch()
    335 
    336     def _fetch_chunk(self):

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in _get_next_batch(self)
    359         # get the first chunks
    360         if self._batch_itr is None:
--> 361             self._fetch_chunk()
    362 
    363         # try to iterate through existing batches

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in _fetch_chunk(self)
    338         if isinstance(chunks, Exception):
    339             self.stop()
--> 340             raise chunks
    341         self._batch_itr = iter(chunks)
    342 

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in load_chunks(self, dev)
    142             if self.dataloader.device != "cpu":
    143                 with self.dataloader._get_device_ctx(dev):
--> 144                     self.chunk_logic(itr)
    145             else:
    146                 self.chunk_logic(itr)

~/anaconda/envs/thesis/lib/python3.7/contextlib.py in inner(*args, **kwds)
     72         def inner(*args, **kwds):
     73             with self._recreate_cm():
---> 74                 return func(*args, **kwds)
     75         return inner
     76 

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in chunk_logic(self, itr)
    124 
    125             if len(chunks) > 0:
--> 126                 chunks = self.dataloader.make_tensors(chunks, self.dataloader._use_nnz)
    127                 # put returns True if buffer is stopped before
    128                 # packet can be put in queue. Keeps us from

~/anaconda/envs/thesis/lib/python3.7/contextlib.py in inner(*args, **kwds)
     72         def inner(*args, **kwds):
     73             with self._recreate_cm():
---> 74                 return func(*args, **kwds)
     75         return inner
     76 

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in make_tensors(self, gdf, use_nnz)
    466 
    467                 batches[n].append(c)
--> 468         return [self._handle_tensors(*batch) for batch in batches]
    469 
    470     def _get_segment_lengths(self, num_samples):

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in <listcomp>(.0)
    466 
    467                 batches[n].append(c)
--> 468         return [self._handle_tensors(*batch) for batch in batches]
    469 
    470     def _get_segment_lengths(self, num_samples):

~/anaconda/envs/thesis/lib/python3.7/contextlib.py in inner(*args, **kwds)
     72         def inner(*args, **kwds):
     73             with self._recreate_cm():
---> 74                 return func(*args, **kwds)
     75         return inner
     76 

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in _handle_tensors(self, cats, conts, labels)
    598                         f"Did not convert {column_name} to sparse due to missing sparse_max entry"
    599                     )
--> 600                 X[column_name] = self._to_sparse_tensor(X[column_name], column_name)
    601 
    602         # TODO: use dict for labels as well?

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/backend.py in _to_sparse_tensor(self, values_offset, column_name)
    493                 + f"largest sequence in this batch have {max_seq_len} length"
    494             )
--> 495         return self._build_sparse_tensor(values, offsets, diff_offsets, num_rows, seq_limit)
    496 
    497     def _to_tensor(self, gdf, dtype=None):

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/torch.py in _build_sparse_tensor(self, values, offsets, diff_offsets, num_rows, seq_limit)
    177 
    178     def _build_sparse_tensor(self, values, offsets, diff_offsets, num_rows, seq_limit):
--> 179         indices = self._get_indices(offsets, diff_offsets)
    180         return self._get_sparse_tensor(values, indices, num_rows, seq_limit)
    181 

~/anaconda/envs/thesis/lib/python3.7/site-packages/nvtabular/loader/torch.py in _get_indices(self, offsets, diff_offsets)
    162     def _get_indices(self, offsets, diff_offsets):
    163         row_ids = torch.arange(len(offsets) - 1, device=self.device)
--> 164         row_ids_repeated = torch.repeat_interleave(row_ids, diff_offsets)
    165         row_offset_repeated = torch.repeat_interleave(offsets[:-1], diff_offsets)
    166         col_ids = torch.arange(len(row_offset_repeated), device=self.device) - row_offset_repeated

RuntimeError: CUDA error: invalid device ordinal
```

Even when I fix the above error manually by fixing the code, I get another error like below with the same script.
```python
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_9524/2595200145.py in <module>
     16     trainer.train_dataset_or_path = train_paths
     17     trainer.reset_lr_scheduler()
---> 18     trainer.train()
     19     trainer.state.global_step +=1
     20     print('finished')

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   1314                         tr_loss_step = self.training_step(model, inputs)
   1315                 else:
-> 1316                     tr_loss_step = self.training_step(model, inputs)
   1317 
   1318                 if (

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers/trainer.py in training_step(self, model, inputs)
   1847                 loss = self.compute_loss(model, inputs)
   1848         else:
-> 1849             loss = self.compute_loss(model, inputs)
   1850 
   1851         if self.args.n_gpu > 1:

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers/trainer.py in compute_loss(self, model, inputs, return_outputs)
   1879         else:
   1880             labels = None
-> 1881         outputs = model(**inputs)
   1882         # Save past state if it exists
   1883         # TODO: this needs to be fixed and made cleaner later.

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/trainer.py in forward(self, *args, **kwargs)
    785     def forward(self, *args, **kwargs):
    786         inputs = kwargs
--> 787         return self.module(inputs, *args)

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/model/base.py in forward(self, inputs, training, **kwargs)
    520         outputs = {}
    521         for head in self.heads:
--> 522             outputs.update(head(inputs, call_body=True, training=training, always_output_dict=True))
    523 
    524         if len(outputs) == 1:

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/model/base.py in forward(self, body_outputs, training, call_body, always_output_dict, **kwargs)
    386 
    387         if call_body:
--> 388             body_outputs = self.body(body_outputs, training=training)
    389 
    390         for name, task in self.prediction_task_dict.items():

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/config/schema.py in __call__(self, *args, **kwargs)
     48         self.check_schema()
     49 
---> 50         return super().__call__(*args, **kwargs)
     51 
     52     def _maybe_set_schema(self, input, schema):

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/block/base.py in forward(self, input, training, **kwargs)
    149 
    150             elif "training" in inspect.signature(module.forward).parameters:
--> 151                 input = module(input, training=training)
    152             else:
    153                 input = module(input)

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/config/schema.py in __call__(self, *args, **kwargs)
     48         self.check_schema()
     49 
---> 50         return super().__call__(*args, **kwargs)
     51 
     52     def _maybe_set_schema(self, input, schema):

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/tabular/base.py in __call__(self, inputs, pre, post, merge_with, aggregation, *args, **kwargs)
    379 
    380         # This will call the `forward` method implemented by the super class.
--> 381         outputs = super().__call__(inputs, *args, **kwargs)  # noqa
    382 
    383         if isinstance(outputs, dict):

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/features/sequence.py in forward(self, inputs, training, **kwargs)
    259         if self.masking:
    260             outputs = self.masking(
--> 261                 outputs, item_ids=self.to_merge["categorical_module"].item_seq, training=training
    262             )
    263 

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/config/schema.py in __call__(self, *args, **kwargs)
     48         self.check_schema()
     49 
---> 50         return super().__call__(*args, **kwargs)
     51 
     52     def _maybe_set_schema(self, input, schema):

~/anaconda/envs/thesis/lib/python3.7/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
    725             result = self._slow_forward(*input, **kwargs)
    726         else:
--> 727             result = self.forward(*input, **kwargs)
    728         for hook in itertools.chain(
    729                 _global_forward_hooks.values(),

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/masking.py in forward(self, inputs, item_ids, training)
    197 
    198     def forward(self, inputs: torch.Tensor, item_ids: torch.Tensor, training=False) -> torch.Tensor:
--> 199         _ = self.compute_masked_targets(item_ids=item_ids, training=training)
    200         if self.mask_schema is None:
    201             raise ValueError("`mask_schema must be set.`")

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/masking.py in compute_masked_targets(self, item_ids, training)
    138         """
    139         assert item_ids.ndim == 2, "`item_ids` must have 2 dimensions."
--> 140         masking_info = self._compute_masked_targets(item_ids, training=training)
    141         self.mask_schema, self.masked_targets = masking_info.schema, masking_info.targets
    142 

~/anaconda/envs/thesis/lib/python3.7/site-packages/transformers4rec/torch/masking.py in _compute_masked_targets(self, item_ids, training)
    381 
    382             labels_to_unmask = torch.masked_select(
--> 383                 sampled_labels_to_unmask, sequences_with_only_labels
    384             )
    385             rows_to_unmask = torch.masked_select(rows_ids, sequences_with_only_labels)

RuntimeError: invalid shape dimension -1095467911
```
