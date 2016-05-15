"""
H2A Index

This module contains classes, functions, objects for: Web service API
They allow Restful querying of the Index
"""

from functools import wraps

import bokeh.charts
import bokeh.embed
import bokeh.plotting
import bokeh.resources
import flask
import irc.config
import irc.evaluation
import irc.utils
import os.path

STATIC_FOLDER = os.path.join(irc.config.PROJECT_BASE, 'static')
TEMPLATES_FOLDER = os.path.join(irc.config.PROJECT_BASE, 'templates')


def generate_chart(query):
    data = dict(precision=[x for x in query.evaluation['precision']],
                recall=[x for x in query.evaluation['recall']])

    line = bokeh.charts.TimeSeries(data, y=['precision'], x='recall',
                                   color=['precision'], title=str(query),
                                   ylabel='Precision', legend=True)

    script, div = bokeh.embed.components(line)

    return {'div': div, 'script': script}


app = flask.Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATES_FOLDER)


def templated(template=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            template_name = template
            if template_name is None:
                template_name = flask.request.endpoint \
                                    .replace('.', '/') + '.html'
            ctx = f(*args, **kwargs)
            if ctx is None:
                ctx = {}
            elif not isinstance(ctx, dict):
                return ctx
            return flask.render_template(template_name, **ctx)

        return decorated_function

    return decorator


@app.route('/', methods=['GET'], endpoint='index')
def index():
    return flask.render_template('index.html')


@app.route('/evaluation', methods=['GET', 'POST'], endpoint='evaluate')
def evaluate():

    if flask.request.method == 'GET':

        return flask.render_template('choose-index.html')
    else:
        idx = flask.request.form['index']
        corpus = irc.utils.read_corpus_from_file(irc.config.FILES['corpus'])
        index = irc.evaluation.model(idx)(corpus)

        queries = irc.evaluation.evaluate_index(index)

        tags = {query.id: generate_chart(query) for query in queries}
        context = {'index': flask.request.form['index'], 'queries': queries, 'tags': tags}

        return flask.render_template('evaluation.html', **context)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
