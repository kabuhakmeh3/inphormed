from flask import request, render_template
from inphormed import app
from inphormed import policy_pull

@app.route('/')

@app.route('/index')
def index():
    user = {'nickname':'inPhormed',
            'url':'http://www.seriously.com/privacy-notice/'}
    return render_template("index.html", title = 'Home', user=user)

@app.route('/', methods=['POST'])
def my_form_post():
    url = request.form['url']
    violated = policy_pull.main(url=url)
    if violated:
        result = 'VIOLATES'
    else:
        result = 'does not violate'
    return render_template('result.html',
                            policy=result)
