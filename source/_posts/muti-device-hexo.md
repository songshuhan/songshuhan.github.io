---
title: Muti-Device-Hexo
date: 2022-06-07 16:41:09
mathjax: true
tags:
- tools
- github
- git
categories:
- Tools
---





这篇博客记录一下$hexo$博客的同步管理及迁移过程中所踩的坑，其实说白了就是我要写博客总不能只在一个设备上写吧，这样我走到哪里都要背着个电脑，那岂不是要累死，当然$hexo$的本意也是让我们可以把环境文件放在$github$上托管的，但是$hexo$在搭建好之后只存放了静态文件，关于环境的配置并没有保存，所以，想要在另一台电脑上重新部署就需要使用$github$进行同步管理了。

$github$上的$repo$创建了两个分支，$master$保存静态页面，$hexo$用于保存环境的全部文件。

其实有很多博客已经说了过程，可以参考：

- http://xuexuan.site/2021/02/04/hexo%E5%8D%9A%E5%AE%A2%E5%A4%9A%E8%AE%BE%E5%A4%87%E5%90%8C%E6%AD%A5/
- https://www.jianshu.com/p/fceaf373d797

这里稍做总结

## 在旧环境下

- 首先在$github$上新建一个分支，取名为$hexo$，注意新建完的分支是$master$分支的复制，我们希望清空然后只存放$hexo$所有的环境配置文件
- $github$上切换到$hexo$分支，`git clone`仓库到本地。
- 此时本地会多出一个`username.github.io`文件夹，命令行`cd`进去，把`.git`文件夹之外的所有文件都删除（如果你看不到这个文件夹，说明是隐藏了。$windows$下需要右击文件夹内空白处，点选’显示/隐藏 异常文件’，$Mac$下我就不知道了）外的其他文件夹。
- 命令行`git add -A`把工作区的变化（包括已删除的文件）提交到暂存区（$ps$:`git add .`提交的变化不包括已删除的文件）。
- 命令行`git commint -m "some description"`提交。
- 命令行`git push origin hexo`推送到远程$hexo$分支。此时刷下$github$，如果正常操作，$hexo$分支应该已经被清空了。
- 复制本地`username.github.io`文件夹中的`.git`文件夹到$hexo$项目根目录下。此时，$hexo$项目已经变成了和远程$hexo$分支关联的本地仓库了。而`username.github.io`文件夹的使命到此为止，你可以把它删掉，因为我们只是把它作为一个“中转站”的角色。以后每次发布新文章或修改网站样式文件时，`git add . & git commit -m "some description" & git push origin hexo`即可把环境文件推送到$hexo$分支。然后再`hexo g -d`发布网站并推送静态文件到$master$分支。

***

## 在新环境下：

​	这部分应该要简单一点，如果你已经搭建过一个$hexo$博客的话。

- 新电脑上安装$node.js$和$git$。
- 安装$hexo$：`npm install -g hexo-cli`。
- $clone$远程仓库到本地 `git clone git@github.com:username/username.github.io.git`。
- 使用`npm install`安装依赖。**<font color=DarkViolet>（这里要注意，看下面的踩坑问题分析）</font>**
- 本地生成网站并开启博客服务器：`hexo g & hexo s`。如果一切正常，此时打开浏览器输入`http://localhost:4000/`已经可以看到博客正常运行了。

***

## 两台电脑上的操作

至此，迁移工作已完成，在两台电脑之间的同步操作如下：

- `git pull`从远程$hexo$分支拉取最新的环境文件到本地，可以理解为$svn$的更新操作。比如在公司写了博客，回家在电脑上也要写需要先执行这一步操作。
- 文章写完，要发布时，需要先提交环境文件，再发布文章。按以下顺序执行命令：`git add .`、`git commit -m "some descrption"`、`git push origin hexo`、`hexo g -d`。

***



其实上面的过程千篇一律，下面要记录一下有价值的关键问题

我们发现在把本地的环境上传到$github$上的时候并没有$theme$文件夹和$node \underline{} modules$文件夹，这就导致了我们在新环境`git pull`的时候其实并不能使用我们的主题，会报错`warn no layout ...`，而且我们在旧环境下辛辛苦苦安装的一些支持$latex$的包，好不容易解决的$hexo$中不能显示图片的问题，这些统统都在文件夹里，$node \underline{} modules$有很多的调整。我们其实玩$hexo$无非就是主题和各种格式的配置费心费力，那这最需要的两个文件夹都拿不到还玩啥，所以我们就要考虑为什么没有把这两个文件夹的内容push上去?

- 远程$hexo$分支中的主题文件夹也为空。原来是在搭建博客开始使用主题时，文件夹是从远程`clone`过来的。所以$theme$文件夹中的`.git`文件夹会导致$theme$文件夹无法被跟踪。这个`.git`文件夹删除之后重新提交。但是，这样解决貌似依然不行，**<font color=DarkViolet>其实根本原因是因为之前主题是设置成被$ignore$,解决方法是`git rm -r --cached cactus`，取消对$cactus$之前的追踪记录。</font>**
- 关于$node \underline{} modules$文件夹，在`git pull`后的新环境下使用`npm install`安装依赖，就会自动生成一个，$node \underline{} modules$但是这个文件夹和我们老环境中的天差地别，之前我们可是好好的进行过调整和修改，所以其实不需要执行这个命令，没有意义，我们只想要老的文件夹，这里主要是因为在$.gitignore$文件中规定了其中的一些文件夹都是被忽略的，是不会被提交的，在`hexo g -d`时部署到$github$上的只有静态文件，而提交环境文件时则不会提交，节省了空间，所以为了能让文件夹能$node \underline{} modules$够成功上传到$github$上，我们需要把$.gitignore$文件中的删除就可以了$node \underline{} modules$

***

最后总结一下在hexo中对数学公式和图片无法显示问题和对标签引入的处理，可以参考下面几个比较直击痛点的博文，现在的垃圾信息，垃圾文章太多了，很浪费时间，要学会甄别

- https://slime0.github.io/2020/07/11/%E8%A7%A3%E5%86%B3%E4%BD%BF%E7%94%A8Typora%EF%BC%8CHexo%E5%8D%9A%E5%AE%A2%E7%9A%84%E5%9B%BE%E7%89%87%E6%97%A0%E6%B3%95%E6%98%BE%E7%A4%BA%E7%9A%84%E5%9D%91/
- http://computetechnologydaily.com:5000/2021/02/12/08055639c3a24350a200247af3d9d524/
- https://www.jianshu.com/p/7ab21c7f0674
