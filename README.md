# VGG16
The project use pretreatment model VGG16(save in *.npy format) to extract feature of input image(you like stylel,et A).
and use VGG16 to extract content of other input image(you like content,et B).
then add or fuse a new picture.(new = A + B)

Focus:
1. "vgg16.npy" you can get it vai Baidu. befer you run ,you need updown.
2. "tf_gfile.py", just delete project result.
3."VGG_content.py", it can help you to analyze "vgg16.npy" content.

Run:
You can just run "sytle_tranfer.py" , the project will produce a new folder nameed "run_style_tranfer". You can open it, and view the result. Certainly, if you consider If you don't think it has what you want, try changing the style to a higher level and the content to a lower level.Besides that, you can increase the number of trainning.

