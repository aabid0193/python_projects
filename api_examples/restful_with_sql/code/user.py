import sqlite3
from flask_restful import Resource, reqparse

class User:
    
    def __init__(self, _id, username, password):
        self.id = _id
        self.username = username
        self.password = password

    # mappings on username
    @classmethod
    def find_by_username(cls, username):  #using current class instead of hardcoding class name
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()

        query = "SELECT * FROM users WHERE username = ?"  # only select where username matches a parameter
        result = cursor.execute(query, (username,)) # parameter must always be in the form of a tuple
        row = result.fetchone()
        if row:
            # row[0] is id, row[1] is username, row[2] is password
            # user = cls(row[0], row[1], row[2])  #if row is not none, return informaiton matching parameters in init method
            user = cls(*row) # pass row as set of positional arguments similar to comment above
        else:
            user = None
        
        connection.close()
        return user
    
    # mappings on id
    @classmethod
    def find_by_id(cls, _id):  #using current class instead of hardcoding class name
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()

        query = "SELECT * FROM users WHERE id = ?"  # only select where username matches a parameter
        result = cursor.execute(query, (_id,)) # parameter must always be in the form of a tuple
        row = result.fetchone()
        if row:
            # row[0] is id, row[1] is username, row[2] is password
            # user = cls(row[0], row[1], row[2])  #if row is not none, return informaiton matching parameters in init method
            user = cls(*row) # pass row as set of positional arguments similar to comment above
        else:
            user = None
        
        connection.close()
        return user

class UserRegister(Resource):

    #parser will parse request to see if requirements are there (username, password)
    parser = reqparse.RequestParser()
    parser.add_argument('username',
                        type = str,
                        required = True,
                        help = "This field cannot be left blank!"
                    )  #terminates request 

    parser.add_argument('password',
                        type = str,
                        required = True,
                        help = "This field cannot be left blank!"
                    )  #terminates request 

    def post(self):
        data = UserRegister.parser.parse_args()

        if User.find_by_username(data['username']):
            return {"message": "A user with that username already exists"}, 400
        
        connection = sqlite3.connect('data.db')
        cursor = connection.cursor()

        query = "INSERT INTO users VALUES (NULL, ?, ?)" ## Null is entered since id is autoincremented
        cursor.execute(query, (data['username'], data['password'])) # type for username and password
        
        connection.commit()
        connection.close()

        return {"message": "User created successfully"}, 201  # 201 is code for created