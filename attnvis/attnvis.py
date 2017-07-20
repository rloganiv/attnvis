"""Attention visualization application."""

import os
import flask

from attnvis import util

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

    # ---DEMO CODE---

    # Preprocess data
    title = title.split()
    description = description.split()

    # Run text attention model
    n = len(title) + len(description)
    import numpy as np
    attention = np.random.rand(n)
    attention = attention / np.sum(attention)

    # Generate highlight info
    title_spans, description_spans = util.generate_spans(
        title=title,
        description=description,
        attention=attention)

    # ---DEMO CODE---

    return flask.render_template(
        'display_data.html',
        title_spans=title_spans,
        description_spans=description_spans,
        attribute=attribute,
        predictions=[]
    )

