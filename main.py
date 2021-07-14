from flask import Flask,render_template,request
import sys
from static.head import get_dataset
app = Flask(__name__,template_folder='templates')

@app.route("/",methods=["GET","POST"])
def index():
    # if request.method == "POST":
    #     p = request.form.get('p')
    #     d = request.form.get('d')
    #     database = request.form.get('database')
    #     print(p,d,database)

    return render_template("index.html")
@app.route("/data",methods=["GET","POST"])
def dataset():
    if request.method == "POST":
        p = request.form.get('p')
        d = request.form.get('d')
        iteration = request.form.get("iteration")
        database = request.form.get('database')
        context =get_dataset(database,int(p),int(d),int(iteration))
        # print(get_dataset(database,int(p),int(d),int(iteration)))

        return render_template("dataset.html",context=context)

app.run(debug=True)
