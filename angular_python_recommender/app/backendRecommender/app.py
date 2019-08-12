from flask import Flask
import json
from modules.Recommender import getRecommendedItems

app = Flask(__name__)
@app.route("/predictions/<int:uid>", strict_slashes=False)
def predictions(uid):
    return json.dumps(getRecommendedItems(uid), indent=2)

if __name__ == '__main__':
    app.run(debug=True)