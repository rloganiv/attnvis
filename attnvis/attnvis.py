import os
import flask


app = flask.Flask(__name__)
app.config.from_object(__name__)


@app.route('/')
def default():
    return flask.render_template('enter_data.html')


@app.route('/render', methods=['POST'])
def render():
    # Obtain attributes
    title = flask.request.form['title']
    description = flask.request.form['description']
    attribute = flask.request.form['attribute']

    # Preprocess data

    # Run text attention model

    # Generate highlight info

    return flask.render_template(
        'display_data.html',
        title=title,
        description=description,
        attribute=attribute,
        predictions=[]
    )

