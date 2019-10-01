# summarize policy

# PICKLE THIS (too messy)
policy_dict = {
        'Contact':['Contact E Mail Address 1stParty',
                   'Contact Phone Number 1stParty','Contact Postal Address 1stParty',
                   'Contact Address Book 1stParty','Contact Password 1stParty',
                   'Contact E Mail Address 3rdParty','Contact 1stParty',
                   'Contact Postal Address 3rdParty','Contact ZIP 1stParty',
                   'Contact Phone Number 3rdParty','Contact City 1stParty',
                   'Contact Password 3rdParty','Contact ZIP 3rdParty',
                   'Contact 3rdParty','Contact City 3rdParty',
                   'Contact Address Book 3rdParty'],

        'Demographic':['Demographic 3rdParty', 'Demographic Age 3rdParty',
                       'Demographic Age 1stParty', 'Demographic 1stParty',
                       'Demographic Gender 1stParty', 'Demographic Gender 3rdParty'],

        'Identifier':['Identifier Cookie or similar Tech 1stParty',
                      'Identifier IP Address 1stParty',
                      'Identifier Cookie or similar Tech 3rdParty',
                      'Identifier Ad ID 1stParty','Identifier Device ID 1stParty',
                      'Identifier IMEI 1stParty','Identifier IMSI 1stParty',
                      'Identifier SSID BSSID 1stParty','Identifier IP Address 3rdParty',
                      'Identifier MAC 1stParty','Identifier 1stParty',
                      'Identifier Mobile Carrier 1stParty','Identifier 3rdParty',
                      'Identifier SIM Serial 1stParty','Identifier Device ID 3rdParty',
                      'Identifier MAC 3rdParty','Identifier Ad ID 3rdParty',
                      'Identifier IMEI 3rdParty','Identifier SIM Serial 3rdParty',
                      'Identifier Mobile Carrier 3rdParty','Identifier IMSI 3rdParty',
                      'Identifier SSID BSSID 3rdParty'],

        'Location':['Location 1stParty','Location Cell Tower 1stParty',
                    'Location IP Address 1stParty','Location 3rdParty',
                    'Location Cell Tower 3rdParty','Location IP Address 3rdParty',
                    'Location Bluetooth 1stParty','Location Bluetooth 3rdParty',
                    'Location GPS 1stParty','Location GPS 3rdParty',
                    'Location WiFi 1stParty','Location WiFi 3rdParty'],

        'Single Sign-On':['Facebook SSO', 'SSO']
        }

icons = {'Performed':'&#10060;','Not Performed':'&#9989;','Not Mentioned':'&#128679;'}

def map_policy(policy_name):
    for general_policy in policy_dict:
        if policy_name in policy_dict[general_policy]:
            return general_policy

def get_type(all_policies):
    '''
    > INPUT
    takes all_policies as Input
    > RETURN
    a summarized version
    overall practice : {}
    '''
    result = all_policies

    for pol in result:
        result[pol]['type'] = map_policy(pol)

    return result

def summarize(all_policies):
    '''
    return a summary (x/y practiced)
    '''
    summary = {}

    for policy in policy_dict:
        tmp = {}
        prac = 0
        not_prac = 0
        no_mention = 0

        for i in all_policies:
            #print(i)
            if all_policies[i]['type'] == policy:
                if all_policies[i]['modality'] == 'Performed':
                    prac += 1
                if all_policies[i]['modality'] == 'Not Performed':
                    not_prac += 1
                if all_policies[i]['modality'] == 'Not Mentioned':
                    no_mention += 1

            #if all_policies[i]['type'] == policy:
            #    if all_policies[i]['modality'] == 'Performed':
            #        prac += 1
            #    if all_policies[i]['modality'] == 'Not Performed':
            #        not_prac += 1
            #    if all_policies[i]['modality'] == 'Not Mentioned':
            #        no_mention += 1

            #if prac > 0:
            #    icon = icons['Performed']
            #elif not_prac > 0:
            #    icon = icons['Not Performed']
            #elif (prac==0) and (not_prac==0):
            #    icon = icons['Not Mentioned']

            #tmp['icon'] = icon
            tmp['practiced'] = prac
            tmp['notpracticed'] = not_prac
            tmp['notmentioned'] = no_mention
            #tmp['policy list'] = policy_dict[pol]

        if prac > 0:
            icon = icons['Performed']
        elif not_prac > 0:
            icon = icons['Not Performed']
        elif (prac==0) and (not_prac==0):
            icon = icons['Not Mentioned']

        tmp['icon'] = icon
        summary[policy] = tmp

    return summary
#if __name__ == '__main__':
#    main()
