from flask import Flask
from flask_restful import Api
from flask_jwt import JWT

from security import authenticate, identity
from user import UserRegister
from item import Item, ItemList

app = Flask(__name__)
app.secret_key = "jose"
api = Api(app)

jwt = JWT(app, authenticate, identity) # /auth is created. This is used for user logins 
# and this creates jwt token to send to identity function to validate

# makes item resouce accessible by api
api.add_resource(Item, '/item/<string:name>') # ex.: http:/127.0.0.1:5000/item/name
api.add_resource(ItemList, '/items')
api.add_resource(UserRegister, '/register')


if __name__ == '__main__':  # only the file that is run is main
    app.run(host='127.0.0.1',
            port=5000,
            debug=True)


#json webtoken (jwt) does obsification of data, or encodes it to pass data back and forth.