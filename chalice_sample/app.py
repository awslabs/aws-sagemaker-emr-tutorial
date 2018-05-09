try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from io import BytesIO
import csv
import sys, os, base64, datetime, hashlib, hmac 
from chalice import Chalice, Response
from chalice import NotFoundError, BadRequestError
import pandas as pd
import sys, os, base64, datetime, hashlib, hmac 
import requests # pip install requests
app = Chalice(app_name='CHALICE_PROJECT')
app.debug = True

try:
    from urlparse import urlparse, parse_qs
except ImportError:
    from urllib.parse import urlparse, parse_qs

import boto3
sagemaker = boto3.client('sagemaker-runtime')

@app.route('/handle_data', methods=['POST'], content_types=['application/x-www-form-urlencoded'])
def handle_data():
    request = app.current_request
    d = parse_qs(app.current_request.raw_body.decode())
    # data to csv
        
    try:
        my_dict = {k:v[0] for k, v in d.iteritems()}
    except AttributeError:
        my_dict = {k:v[0] for k, v in d.items()}
    
    my_dict['content'] = my_dict.pop('raw_text')
    f = StringIO()
    w = csv.DictWriter(f, my_dict.keys())
    w.writeheader()
    w.writerow(my_dict)

    content_type = 'text/csv'
    headers = {'Content-Type':content_type,
               'Accept': 'Accept'}

    res = sagemaker.invoke_endpoint(
                    EndpointName='CHALICE_PROJECT',
                    Body=f.getvalue(),
                    ContentType='text/csv',
                    Accept='Accept'
                )
    # res.json to dict
    # format dict to pretty

    result = res['Body']

    f = StringIO()
    f.write(result.read().decode().replace("\\r\\n", "<br>").replace("\\n", "<br>"))
    f.seek(0)

    result_html = """
<!DOCTYPE html>
<html>
<head>
        <title>Model Prediction</title>
<style>
body{{
        background-color: white;
        text-align: center;
        color: #545b64;
        font-size: 16px;

}}
</style>
</head>
<body>
        <center>{0}</center>
        <table align=center><tr valign=top align=left><td width=28%>{1}</td><td width=28%>{2}</td><td width=28%>{3}</td></tr></table>

</body>
</html>


    """.format(f.readline(), f.readline(), f.readline(), f.readline())

    return Response(body=result_html,
                    status_code=200,
                    headers={'Content-Type': 'text/html'})
