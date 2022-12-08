
## Introduction
In recent days, streaming technology has greatly promoted
the development in the field of livestream. Due to the excessive
length of livestream records, it’s quite essential to extract
highlight segments with the aim of effective reproduction and
redistribution. Although there are lots of approaches proven
to be effective in the highlight detection for other modals, the
challenges existing in livestream processing, such as the extreme
durations, large topic shifts, much irrelevant information
and so forth, heavily hamper the adaptation and compatibility
of these methods. In this paper, we formulate a new task
Livestream Highlight Detection, discuss and analyze the difficulties
listed above and propose a novel architecture **AntPivot**
to solve this problem. Concretely, we first encode the original
data into multiple views and model their temporal relations
to capture clues in a chunked attention mechanism.
Afterwards, we try to convert the detection of highlight clips
into the search for optimal decision sequences and use the
fully integrated representations to predict the final results in
a dynamic-programming mechanism. Furthermore, we construct
a fully-annotated dataset AntHighlight to instantiate
this task and evaluate the performance of our model. The extensive
experiments indicate the effectiveness and validity of
our proposed method.

1. Environment dependency(环境依赖)

    torch

    sentence-transformers

    transformers
 
2. Data preparation(数据准备)
   
   从如下链接中下载训练集(train),验证集合(eval),测试集合(test),放到data目录,目录结构如下:

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

	运行如下命令，生成数据

   ``` python paper/prepro/preprocess_label4.py ```
    

3. Train(训练)

    模型配置文件存放在paper/config 目录
  
     ``` cd paper ```

     ``` python main.py --config ${配置文件名} ```

4. Test(测试)
    
     ``` python main.py --config ${配置文件名} --eval-epoch ${评测模型迭代次数} ```
 

