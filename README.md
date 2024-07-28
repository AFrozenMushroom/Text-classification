# Text-classification-文本分析
基于tensorflow和keras的文本分析模型，可以识别一段评论是积极还是消极
make-model.py是制作模型的程序，包括下载数据，处理数据，构建模型，训练模型和保存模型
运行make-model.py，模型将下载数据集aclImdb，包括解压好的aclImdb文件夹和未解压的tar.gz压缩包
make-model.py会自动剔除数据集中的无用部分，并将文本数据词向量化成张量数据，通过张量进行训练，产生my_model.keras模型文件和准确率
模型本身使用一个Embedding层，一个池化层，一个全连接层组成，其中插入两个dropout层用于防止过拟合，返回一个数字，0表示积极，1表示消极
use-model.py是使用模型的程序，会调用模型my_model.py，并将数据集中的测试集代入并返回结果
运行use-model.py，将随机选择一个数据，返回数据的文本内容（英文），预测结果和正确结果
本模型可以分析自定义的文本，在use_model.py中将想识别的英文文本赋值于变量，将变量代入模型，也会返回结果。请确保输入文本有一定的积极消极态度，过于中立的文本难以识别
如果没有avlImdb数据文件夹，use-model.py无法运行。请先运行make-model.py下载数据，并保证数据文件夹和use-model.py在同一个文件夹中
