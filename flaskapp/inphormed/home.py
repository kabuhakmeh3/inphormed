from flask import request, render_template
from inphormed import app
#from inphormed import policy_pull
#from inphormed import evaluate_policy
import pickle
from inphormed import evaluate_policy_dynamic
from inphormed import summarize_policy

#list_file = 'policy_list.pckl'
list_file = 'policy_list_ord_all.pckl'
#list_file = 'policy_list_ord_top.pckl' # top 20
policy_tmp = pickle.load(open('inphormed/static/pickles/'+list_file,'rb'))
policy_list = [p.replace('_',' ') for p in policy_tmp]

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
    # ADD THIS AFTER MORE INPUT CAN BE HANDLED
    #practices = request.form['practices']
    #chosen_policies = []
    #for a in practices:
    #   for b in policy_list:
    #       if b.casefold().contains(a.casefold())
    #           chosen_policies.append(b)
    #tmp_result = evaluate_policy.main(url=url)
    tmp_result, tmp_sentences = evaluate_policy_dynamic.main(url=url)
    icons = {'Performed':'&#10060;','Not Performed':'&#9989;','Not Mentioned':'&#128679;'}
    result = {}

    # ADD:
    # option to take input for TYPES of policies user is interested if __name__ == '__main__':
    # show results for policies of interest (Location, Contacts, Media, etc)

    #for pol in chosen_policies:
    for pol in policy_list:
        if pol in tmp_result:
            tmp_dict = {}
            tmp_dict['modality'] = tmp_result[pol]
            tmp_dict['sentences'] = tmp_sentences[pol]
            tmp_dict['icon'] = icons[tmp_result[pol]]
            result[pol] = tmp_dict
        else:
            tmp_dict = {}
            tmp_dict['modality'] = 'Not Mentioned'
            #tmp_dict['sentences'] = ['This practice is not explicitly mentioned in the policy']
            tmp_dict['sentences'] = None
            tmp_dict['icon'] = icons['Not Mentioned']
            result[pol] = tmp_dict

    # this is new
    #result = summarize_policy.main(all_policies=result)
    result = summarize_policy.get_type(all_policies=result)

    summarized_result = summarize_policy.summarize(result)

    clean_result = {}
    for prac in result:
        tmp = {}
        if result[prac]['sentences']:
            raw_sentences = result[prac]['sentences']
            #clean_sentences = '\n'.join(s for s in raw_sentences)
            #tmp['sentences'] = clean_sentences
            tmp['sentences'] =  '\n'.join(s for s in raw_sentences)
            tmp['modality'] = result[prac]['modality']
            clean_result[prac] = tmp
    #result = summarized_result

    return render_template('result.html', policy=clean_result,
                            summary=summarized_result)
