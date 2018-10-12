import json

path = 'D:\python_projects\ServeNet_others\RCNN_acc_category.json'

with open(path, 'r') as f:
    array = json.load(f)

print(array)

labels = ["Tools","Financial","Messaging","eCommerce","Payments","Social","Enterprise","Mapping","Telephony","Science",
          "Government","Email","Security","Reference","Video","Travel","Sports","Search","Advertising","Transportation",
          "Education","Games","Music","Photos","Cloud","Bitcoin","Project Management","Data","Backend","Database",
          "Shipping","Weather","Application Development","Analytics","Internet of Things","Medical","Real Estate",
          "Events","Banking","Stocks","Entertainment","Storage","Marketing","File Sharing","News Services","Domains",
          "Chat","Media","Images","Other"
]


for label in labels:
    value = array[label]
    print('%.2f' % (value * 100))
