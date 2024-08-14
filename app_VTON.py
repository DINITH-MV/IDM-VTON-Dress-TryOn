from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    # Example data
    user_data = {
        "id": 1,
        "name": "Rohan Kokkula",
        "email": "rohan@example.com",
        "age": 24
    }
    # Return data as JSON
    return jsonify(user_data)

if __name__ == '__main__':
    app.run(debug=True, port=3000)