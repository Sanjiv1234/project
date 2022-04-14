from flask import Flask,render_template,request
import pickle

app = Flask(__name__,template_folder='template')

import caption_it

@app.route('/')
def home():
	return render_template("index.html")


@app.route('/',methods = ["POST"])
def home2():
	if request.method == "POST":
	  
		user = request.files["userfile"]

		path = "./static/{}".format(user.filename)
		user.save(path)
		
		caption = caption_it.caption_the_img(path)
		audio= caption_it.text_to_audio(path)
		result_dic = {
		'image':path,
		'cap':caption,
		'audio':audio
        }
	return render_template("index.html",your_result = result_dic)



if __name__ == '__main__':
    app.run(debug = True)