from flask import request, render_template
from inphormed import app
#from inphormed import policy_pull
from inphormed import evaluate_policy

@app.route('/')
def index():
    user = {'nickname':'inphormed',
            'url':'http://www.seriously.com/privacy-notice/'}
    return render_template('index.html', title = 'Home', user=user)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/', methods=['POST'])
def result():
    url = request.form['url']
    tmp_result = evaluate_policy.main(url=url)
    icons = {'Performed':'&#10060;','Not Performed':'&#9989;'}
    result = {}
    for pol in tmp_result:
        tmp_dict = {}
        tmp_dict['modality'] = tmp_result[pol]
        tmp_dict['icon'] = icons[tmp_result[pol]]
        result[pol] = tmp_dict
        
    return render_template('result.html',
                            policy=result)
