
## 说明 

- 项目背景：说明创建本项目的背景与动机，创建本项目试图解决的问题 
- 安装方法：说明如何快速上手使用该项目
- 使用方法：列出本项目能够提供的功能以及使用这些功能的方法

1. 环境依赖

    torch

    sentence-transformers

    transformers
 
2. 数据准备
   
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
    

3. 训练

    模型配置文件存放在paper/config 目录
  
     ``` cd paper ```

     ``` python main.py --config ${配置文件名} ```

4. 测试
    
     ``` python main.py --config ${配置文件名} --eval-epoch ${评测模型迭代次数} ```
 

## 附加内容

视项目的实际情况，同样也应该包含以下内容：

- 项目特性：说明本项目相较于其他同类项目所具有的特性
- 兼容环境：说明本项目能够在什么平台上运行
- 使用示例：展示一些使用本项目的小demo
- 主要项目负责人：使用“@”标注出本项目的主要负责人，方便项目的用户沟通
- 参与贡献的方式：规定好其他用户参与本项目并贡献代码的方式
- 项目的参与者：列出项目主要的参与人
- 已知用户：列出已经在生产环境中使用了本项目的全部或部分组件的公司或组织
- 赞助者：列出为本项目提供赞助的用户

