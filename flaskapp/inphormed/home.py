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

    policies = {'Contact_E_Mail_Address_1stParty':'email',
                'Location_1stParty':'location',
                'Identifier_Cookie_or_similar_Tech_3rdParty':'cookies',
                'Contact_Phone_Number_1stParty':'phone',
                'SSO':'sso'}
    result = {}
    for pol in policies:
        if violated:
            tmp = 'VIOLATES'
        else:
            tmp = 'does not violate'
        result[policies[pol]] = tmp

    return render_template('result.html',
                            policy=result)
