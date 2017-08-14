"""Attention visualization application."""

import os
import pickle
import flask
import tensorflow as tf
from mumie.utils.preprocess import process_text
from mumie.utils.structs import Lexicon
from mumie.utils.vocabulary import Vocabulary
from mumie.inference import InferenceWrapper

from attnvis import util


app = flask.Flask(__name__)
app.config.from_object(__name__)

# Construct lexicon for mapping words to model inputs
lexicon = Lexicon(
    description_vocab=Vocabulary.from_file(os.path.join(app.root_path, 'static/vocab/desc')),
    attribute_vocab=Vocabulary.from_file(os.path.join(app.root_path, 'static/vocab/attr')),
    value_vocab=Vocabulary.from_file(os.path.join(app.root_path, 'static/vocab/value')))

# Restore model from checkpoint
model = InferenceWrapper()
config_path = os.path.join(app.root_path, 'static/ckpt/config.pkl')
checkpoint_path = os.path.join(app.root_path, 'static/ckpt/model.ckpt-3747')
model.load(config_path, checkpoint_path)

# Add to config
app.config.update({
    'MODEL': model,
    'LEXICON': lexicon
})


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
    lexicon = app.config['LEXICON']
    desc_vocab = lexicon.description_vocab

    title = process_text(title)
    title_check = [desc_vocab.id_to_word(desc_vocab.word_to_id(word)) for word
                   in title]

    description = process_text(description)
    description_check = [desc_vocab.id_to_word(desc_vocab.word_to_id(word)) for word
                   in description]

    desc_feed = [lexicon.description_vocab.word_to_id(word) for
                 word in title + description]
    print(desc_feed)
    attr_feed = lexicon.attribute_vocab.word_to_id(attribute)

    # Run model
    model = app.config['MODEL']
    output, attention = model.predict(desc_feed, attr_feed)
    print(attention)

    # Generate predictions
    predictions = []
    for result in output.argsort()[-5:][::-1]:
        try:
            prediction = {
                'value': lexicon.value_vocab.id_to_word(result),
                'prob': output[result]
            }
        except:
            pass
        predictions.append(prediction)

    # Generate highlight info
    title_spans, description_spans = util.generate_spans(
        title=title_check,
        description=description_check,
        attention=attention)

    return flask.render_template(
        'display_data.html',
        title_spans=title_spans,
        description_spans=description_spans,
        attribute=attribute,
        predictions=predictions
    )

