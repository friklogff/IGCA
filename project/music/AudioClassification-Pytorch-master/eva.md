D:\AN\envs\yolov11\python.exe D:\ultralytics-main\zzzaaa_project\music\AudioClassification-Pytorch-master\eval.py 
2025-01-19 21:01:22.935 | INFO     | macls.utils.utils:print_arguments:12 - ----------- 额外配置参数 -----------
2025-01-19 21:01:22.935 | INFO     | macls.utils.utils:print_arguments:14 - configs: configs/cam++.yml
2025-01-19 21:01:22.935 | INFO     | macls.utils.utils:print_arguments:14 - resume_model: models/CAMPPlus_Fbank/best_model/
2025-01-19 21:01:22.935 | INFO     | macls.utils.utils:print_arguments:14 - save_matrix_path: output/images/
2025-01-19 21:01:22.935 | INFO     | macls.utils.utils:print_arguments:14 - use_gpu: True
2025-01-19 21:01:22.935 | INFO     | macls.utils.utils:print_arguments:15 - ------------------------------------------------
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:18 - ----------- 配置文件参数 -----------
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:21 - dataset_conf:
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:24 - 	dataLoader:
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		batch_size: 64
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		drop_last: True
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		num_workers: 8
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:24 - 	dataset:
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		max_duration: 3
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		min_duration: 0.4
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		sample_rate: 16000
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		target_dB: -20
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		use_dB_normalization: True
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:24 - 	eval_conf:
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		batch_size: 8
2025-01-19 21:01:22.970 | INFO     | macls.utils.utils:print_arguments:26 - 		max_duration: 20
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:28 - 	label_list_path: dataset/label_list.txt
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:28 - 	test_list: dataset/test_list.txt
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:28 - 	train_list: dataset/train_list.txt
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:21 - model_conf:
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:28 - 	model: CAMPPlus
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:24 - 	model_args:
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:26 - 		num_class: None
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:21 - optimizer_conf:
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:28 - 	optimizer: Adam
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:24 - 	optimizer_args:
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:26 - 		lr: 0.001
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:26 - 		weight_decay: 1e-05
2025-01-19 21:01:22.971 | INFO     | macls.utils.utils:print_arguments:28 - 	scheduler: WarmupCosineSchedulerLR
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:24 - 	scheduler_args:
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:26 - 		max_lr: 0.001
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:26 - 		min_lr: 1e-05
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:26 - 		warmup_epoch: 5
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:21 - preprocess_conf:
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:28 - 	feature_method: Fbank
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:24 - 	method_args:
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:26 - 		num_mel_bins: 80
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:26 - 		sample_frequency: 16000
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:28 - 	use_hf_model: False
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:21 - train_conf:
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:28 - 	enable_amp: False
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:28 - 	label_smoothing: 0.0
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:28 - 	log_interval: 10
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:28 - 	max_epoch: 60
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:28 - 	use_compile: False
2025-01-19 21:01:22.972 | INFO     | macls.utils.utils:print_arguments:31 - ------------------------------------------------
2025-01-19 21:01:22.972 | WARNING  | macls.trainer:__init__:70 - Windows系统不支持多线程读取数据，已自动关闭！
2025-01-19 21:01:22.973 | INFO     | macls.data_utils.featurizer:__init__:51 - 使用【Fbank】提取特征
对列表[dataset/test_list.txt]进行长度排序: 100%|██████████| 874/874 [00:03<00:00, 239.26it/s]
2025-01-19 21:01:26.851 | INFO     | macls.models:build_model:20 - 成功创建模型：CAMPPlus，参数为：{'num_class': 10}
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
CAMPPlus                                      [1, 10]                   --
├─FCM: 1-1                                    [1, 320, 98]              --
│    └─Conv2d: 2-1                            [1, 32, 80, 98]           288
│    └─BatchNorm2d: 2-2                       [1, 32, 80, 98]           64
│    └─Sequential: 2-3                        [1, 32, 40, 98]           --
│    │    └─BasicResBlock: 3-1                [1, 32, 40, 98]           19,648
│    │    └─BasicResBlock: 3-2                [1, 32, 40, 98]           18,560
│    └─Sequential: 2-4                        [1, 32, 20, 98]           --
│    │    └─BasicResBlock: 3-3                [1, 32, 20, 98]           19,648
│    │    └─BasicResBlock: 3-4                [1, 32, 20, 98]           18,560
│    └─Conv2d: 2-5                            [1, 32, 10, 98]           9,216
│    └─BatchNorm2d: 2-6                       [1, 32, 10, 98]           64
├─Sequential: 1-2                             [1, 512]                  --
│    └─TDNNLayer: 2-7                         [1, 128, 49]              --
│    │    └─Conv1d: 3-5                       [1, 128, 49]              204,800
│    │    └─Sequential: 3-6                   [1, 128, 49]              256
│    └─CAMDenseTDNNBlock: 2-8                 [1, 512, 49]              --
│    │    └─CAMDenseTDNNLayer: 3-7            [1, 32, 49]               39,520
│    │    └─CAMDenseTDNNLayer: 3-8            [1, 32, 49]               43,680
│    │    └─CAMDenseTDNNLayer: 3-9            [1, 32, 49]               47,840
│    │    └─CAMDenseTDNNLayer: 3-10           [1, 32, 49]               52,000
│    │    └─CAMDenseTDNNLayer: 3-11           [1, 32, 49]               56,160
│    │    └─CAMDenseTDNNLayer: 3-12           [1, 32, 49]               60,320
│    │    └─CAMDenseTDNNLayer: 3-13           [1, 32, 49]               64,480
│    │    └─CAMDenseTDNNLayer: 3-14           [1, 32, 49]               68,640
│    │    └─CAMDenseTDNNLayer: 3-15           [1, 32, 49]               72,800
│    │    └─CAMDenseTDNNLayer: 3-16           [1, 32, 49]               76,960
│    │    └─CAMDenseTDNNLayer: 3-17           [1, 32, 49]               81,120
│    │    └─CAMDenseTDNNLayer: 3-18           [1, 32, 49]               85,280
│    └─TransitLayer: 2-9                      [1, 256, 49]              --
│    │    └─Sequential: 3-19                  [1, 512, 49]              1,024
│    │    └─Conv1d: 3-20                      [1, 256, 49]              131,072
│    └─CAMDenseTDNNBlock: 2-10                [1, 1024, 49]             --
│    │    └─CAMDenseTDNNLayer: 3-21           [1, 32, 49]               56,160
│    │    └─CAMDenseTDNNLayer: 3-22           [1, 32, 49]               60,320
│    │    └─CAMDenseTDNNLayer: 3-23           [1, 32, 49]               64,480
│    │    └─CAMDenseTDNNLayer: 3-24           [1, 32, 49]               68,640
│    │    └─CAMDenseTDNNLayer: 3-25           [1, 32, 49]               72,800
│    │    └─CAMDenseTDNNLayer: 3-26           [1, 32, 49]               76,960
│    │    └─CAMDenseTDNNLayer: 3-27           [1, 32, 49]               81,120
│    │    └─CAMDenseTDNNLayer: 3-28           [1, 32, 49]               85,280
│    │    └─CAMDenseTDNNLayer: 3-29           [1, 32, 49]               89,440
│    │    └─CAMDenseTDNNLayer: 3-30           [1, 32, 49]               93,600
│    │    └─CAMDenseTDNNLayer: 3-31           [1, 32, 49]               97,760
│    │    └─CAMDenseTDNNLayer: 3-32           [1, 32, 49]               101,920
│    │    └─CAMDenseTDNNLayer: 3-33           [1, 32, 49]               106,080
│    │    └─CAMDenseTDNNLayer: 3-34           [1, 32, 49]               110,240
│    │    └─CAMDenseTDNNLayer: 3-35           [1, 32, 49]               114,400
│    │    └─CAMDenseTDNNLayer: 3-36           [1, 32, 49]               118,560
│    │    └─CAMDenseTDNNLayer: 3-37           [1, 32, 49]               122,720
│    │    └─CAMDenseTDNNLayer: 3-38           [1, 32, 49]               126,880
│    │    └─CAMDenseTDNNLayer: 3-39           [1, 32, 49]               131,040
│    │    └─CAMDenseTDNNLayer: 3-40           [1, 32, 49]               135,200
│    │    └─CAMDenseTDNNLayer: 3-41           [1, 32, 49]               139,360
│    │    └─CAMDenseTDNNLayer: 3-42           [1, 32, 49]               143,520
│    │    └─CAMDenseTDNNLayer: 3-43           [1, 32, 49]               147,680
│    │    └─CAMDenseTDNNLayer: 3-44           [1, 32, 49]               151,840
│    └─TransitLayer: 2-11                     [1, 512, 49]              --
│    │    └─Sequential: 3-45                  [1, 1024, 49]             2,048
│    │    └─Conv1d: 3-46                      [1, 512, 49]              524,288
│    └─CAMDenseTDNNBlock: 2-12                [1, 1024, 49]             --
│    │    └─CAMDenseTDNNLayer: 3-47           [1, 32, 49]               89,440
│    │    └─CAMDenseTDNNLayer: 3-48           [1, 32, 49]               93,600
│    │    └─CAMDenseTDNNLayer: 3-49           [1, 32, 49]               97,760
│    │    └─CAMDenseTDNNLayer: 3-50           [1, 32, 49]               101,920
│    │    └─CAMDenseTDNNLayer: 3-51           [1, 32, 49]               106,080
│    │    └─CAMDenseTDNNLayer: 3-52           [1, 32, 49]               110,240
│    │    └─CAMDenseTDNNLayer: 3-53           [1, 32, 49]               114,400
│    │    └─CAMDenseTDNNLayer: 3-54           [1, 32, 49]               118,560
│    │    └─CAMDenseTDNNLayer: 3-55           [1, 32, 49]               122,720
│    │    └─CAMDenseTDNNLayer: 3-56           [1, 32, 49]               126,880
│    │    └─CAMDenseTDNNLayer: 3-57           [1, 32, 49]               131,040
│    │    └─CAMDenseTDNNLayer: 3-58           [1, 32, 49]               135,200
│    │    └─CAMDenseTDNNLayer: 3-59           [1, 32, 49]               139,360
│    │    └─CAMDenseTDNNLayer: 3-60           [1, 32, 49]               143,520
│    │    └─CAMDenseTDNNLayer: 3-61           [1, 32, 49]               147,680
│    │    └─CAMDenseTDNNLayer: 3-62           [1, 32, 49]               151,840
│    └─TransitLayer: 2-13                     [1, 512, 49]              --
│    │    └─Sequential: 3-63                  [1, 1024, 49]             2,048
│    │    └─Conv1d: 3-64                      [1, 512, 49]              524,288
│    └─Sequential: 2-14                       [1, 512, 49]              --
│    │    └─BatchNorm1d: 3-65                 [1, 512, 49]              1,024
│    │    └─ReLU: 3-66                        [1, 512, 49]              --
│    └─StatsPool: 2-15                        [1, 1024]                 --
│    └─DenseLayer: 2-16                       [1, 512]                  --
│    │    └─Conv1d: 3-67                      [1, 512, 1]               524,288
│    │    └─Sequential: 3-68                  [1, 512]                  --
├─Linear: 1-3                                 [1, 10]                   5,130
===============================================================================================
Total params: 7,181,354
Trainable params: 7,181,354
Non-trainable params: 0
Total mult-adds (M): 552.44
===============================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 41.22
Params size (MB): 28.73
Estimated Total Size (MB): 69.98
===============================================================================================
2025-01-19 21:01:27.739 | INFO     | macls.trainer:evaluate:356 - 成功加载模型：models/CAMPPlus_Fbank/best_model/model.pth
100%|██████████| 110/110 [01:19<00:00,  1.38it/s]
评估消耗时间：85s，loss：0.05881，accuracy：0.98523

Process finished with exit code 0
