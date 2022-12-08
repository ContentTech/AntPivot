
## AntPivot: An Novel Architecture To Dectect Hightlight In Livestream

AntPivot is a novel architecture via chunked attention mechanism for the task of livestream highlight dectection.The main contributions can be summarized in the following aspects:

* we formulate the task of Livestream Highlight Detection to explore the extraction of important livestream segments, which is actually an essential pre-processing step for lots of downstream tasks and be also regarded as a meaningful complement to the research in related areas.

* we propose a novel architecture named AntPivot, which introduces a newly-developed chunked attention module, and devise a special dynamic-programming mechanism to address this problem and serve as a baseline approach in this field.

* we collect a fully annotated dataset AntHighlight from the livestream records in the domain of fortune and insurance and prove the feasibility and effectiveness of our proposed approach on this dataset.

## 1. Environment dependency(环境依赖)

    torch==1.5.0

    sentence-transformers

    transformers==4.9.1
 
## 2. Data preparation(数据准备)
   
   1) 从如下链接中下载训练集(train),验证集合(eval),测试集合(test),放到data目录,目录结构如下:

   url链接文件：https://www.aliyundrive.com/s/8HAwJqwyNUr

	|-- data

		|-- train

			|-- asr

			|-- label

			|-- audio

		|-- eval

			|-- asr

			|--label

			|--audio

		|-- test

			|-- asr

			|--label

			|--audio

	2) 运行如下命令，生成数据

   ``` python paper/prepro/preprocess_label4.py ```
    

## 3. Train(训练)

    模型配置文件存放在paper/config 目录
  
     ``` cd paper ```

     ``` python main.py --config ${配置文件名} ```

## 4. Test(测试)
    
     ``` python main.py --config ${配置文件名} --eval-epoch ${评测模型迭代次数} ```
 

## Citation

if you find our work useful, please consider citing AntPivot

```
    @article{antprivot,
	title={AntPivot: Livestream Highlight Detection via Hierarchical Attention Mechanism},
	author={Yang Zhao, Xuan Lin, Wenqiang Xu, Maozong Zheng, Zhengyong Liu, Zhou Zhao},
	journal={arXiv preprint	arXiv:2206.04888},
        year={2022}
	}

```
