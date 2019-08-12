from werkzeug.security import safe_str_cmp # compares strings easily and accounts for unicodes and encoding problems
from user import User

## The below commented block is user mappings without sql storage of user information, only for storage in memory
#users = [
#    User(1, 'bob', 'asdf')
#]   # user table
#username_mapping = {u.username: u for u in users}  ## assigns key value pairs for sets
#userid_mapping = {u.id: u for u in users}  # dicitonary for id information


#ex user userid_mapping[1]
def authenticate(username, password):
    #user = username_mapping.get(username, None) ## get username value from dict (this is without use of sqlite if storing objects in memory)
    user = User.find_by_username(username)
    if user and safe_str_cmp(user.password, password): #compare strings instead of using '=='
        return user

def identity(payload):
    user_id = payload['identity']
    #return userid_mapping.get(user_id, None)  # only for storing in in-memory db
    return User.find_by_id(user_id)