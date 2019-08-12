from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_jwt import JWT, jwt_required

from security import authenticate, identity

app = Flask(__name__)
app.secret_key = "jose"
api = Api(app)

jwt = JWT(app, authenticate, identity) # /auth is created. This is used for user logins 
# and this creates jwt token to send to identity function to validate


items = []

class Item(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('price',
        type = float,
        required = True,
        help = "This field cannot be left blank!"
    )  #terminates request if field is not there
        
    @jwt_required()
    def get(self, name):
        item = next(filter(lambda x: x['name'] == name, items), None) #get next item each time this is called. Return None if no more items exist
        #return [item for item in items if item['name'] == name]
        return {'item': item}, 200 if item else 404 # status code for not found

    def post(self, name):
        if next(filter(lambda x: x['name']==name, items), None):
            return {'message', f"An item with name '{name}' already exists."}, 400

        data = Item.parser.parse_args()  #error first approach, this should be below filter to make sure errors are clear

        data = request.get_json()
        item = {'name': name, 'price': data['price']}
        items.append(item)
        return item, 201 #status code for created. 202 is code for accepted, if you are delaying creation

    #@jwt_required()
    def delete(self, name):
        global items
        items = list(filter(lambda x: x['name'] != name, items)) # overwrite items list with list without deleted item
        return {'message': 'Item deleted'}

    #@jwt_required()
    def put(self, name):
        data = Item.parser.parse_args()
        item = next(filter(lambda x: x['name'] == name, items), None)
        if item is None:
            item = {'name': name, 'price': data['price']}
            items.append(item)
        else:
            item.update(data)
        return item

class ItemList(Resource):
    def get(self):
        return {'items': items}


# makes item resouce accessible by api
api.add_resource(Item, '/item/<string:name>') # ex.: http:/127.0.0.1:5000/item/name
api.add_resource(ItemList, '/items')

app.run(host='127.0.0.1', port=5000)


#json webtoken (jwt) does obsification of data, or encodes it to pass data back and forth.