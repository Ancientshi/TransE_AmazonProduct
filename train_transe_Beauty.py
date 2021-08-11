import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# 训练数据载入
train_dataloader = TrainDataLoader(
	in_path = "/home/shiyunxiao/TransE/TransE_AmazonProduct/benchmarks/Beauty/",  #数据路径
	nbatches = 200,#每次150条数据
	threads = 12, #8线程
	sampling_mode = "normal", #取样模式？？

	bern_flag = 1, #？？
	filter_flag = 1, #？？
	neg_ent = 25, #负样本实体数量
	neg_rel = 0) #负样本关系数量

# 测试数据载入
test_dataloader = TestDataLoader("/home/shiyunxiao/TransE/TransE_AmazonProduct/benchmarks/Beauty/", "link")

# 定义模型
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(), #获取总实体
	rel_tot = train_dataloader.get_rel_tot(), #获取总关系
	dim = 200, #Embedding维度300
	p_norm = 1, #使用第一范式
	norm_flag = True) #是否使用范式


# 负样本采样
model = NegativeSampling(
	model = transe, #transE模型
	loss = MarginLoss(margin = 5.0), #margin为1
	batch_size = train_dataloader.get_batch_size() #批次数量
)

# 训练模型
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 300, alpha = 0.1, use_gpu = True)
trainer.run()
#存模型
transe.save_checkpoint('/home/shiyunxiao/TransE/TransE_AmazonProduct/checkpoint/Beauty.ckpt')

# 测试模型
#载入模型
transe.load_checkpoint('/home/shiyunxiao/TransE/TransE_AmazonProduct/checkpoint/Beauty.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = True)