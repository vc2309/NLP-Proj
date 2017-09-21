#!/usr/local/bin/python3
# Context Handler of the Chatbot Webhook APP
# Created By Intralogue
from json import loads

from chatbot.helper import debug
from .facebook import FacebookRely
from .search import Search
from . import original


class Context:
    def __init__(self, originalmsg, template, results, model):
        self._model = model
        self._template = template
        self._results = results
        # debug('Incoming context: {}'.format(self.context))
        self.okay = False
        if originalmsg:
            (
                self.message,
                self.book_keyword,
                self.class_probability,
                self.general_slots,
                self.words
            ) = original.parser(originalmsg)
            self.intent = self.class_probability[0]
            self.okay = True

    def process(self):
        if self.okay:
            actions = []
            intent = self.intent['intent']
            self.context = self._model.memory.get('context')
            # 1. check if context content Search or intent is keyword search
            if 'Search' in self.context or 'keyword search' in intent:
                keyword = self.book_keyword or self.message
                search_payload = self._model.get_payload('SEARCH_RESULT')
                self._model.memory['cquery']['refinements'] = [keyword]
                self._model.memory['cquery']['keyword'] = keyword
                self._model.memory['cquery']['refinements'] = list(
                    set(self._model.memory['cquery']['refinements'])
                )
                search = Search(
                    self.message, self.context, self._model.memory['cquery']
                )
                authors, date, elems, stype = search.start()
                search_result = search_payload.get('action')
                # debug(search_result, 'search_result')
                if elems:
                    search_result[0]['text'] = search_result[0]['text'].format(
                        num=len(elems)
                    )
                    search_result.insert(1, {
                        'type': 'search',
                        'elems': elems,
                        'context': 'Search ' + stype
                    })
                else:
                    search_result[0]['text'] = search_result[0]['error']
                self._model.search_result(elems)
                self._model.memory['cquery']['authors'] = authors
                self._model.memory['cquery']['date'] = date

                actions.extend(search_result)
            # 2. if not then check if intent is ask a librarin
            elif intent == 'ask a librarian':
                # set context to Ask Librarin
                self.context = 'Ask Librarin'
                # set result as ask for user input the question to ask librarin
                actions.append({
                    'type': 'text',
                    'text': (
                        'Hi there. '
                        'Please leave a message to a librarian. '
                        'She will reach out to you as soon as possibale.'
                    )
                })
            # 3. if not then check if context is Ask Librarin
            elif 'Ask Librarian' in self.context:
                # if highest intent score is highter then .6
                if float(self.intent.get('score', 0)) > 0.6:
                    # then ask for the intent is correct or not
                    tmp = self._model.get_payload('BUTTON_HELPFUL_CONFIRM')
                    tmp = tmp.get('action', {})
                    tmp[1]['buttons'] = [
                        ['Yes', 'HELPFUL_YES_'],
                        ['No', 'HELPFUL_NO_']
                    ]
                    tmp[1]['postfix'] = True
                    actions.exten(tmp)
                    del tmp
                else:
                    # else reply received message
                    # and set the user to the beginning
                    tmp = self._model.get_payload(
                        'BUTTON_START_GENERAL_ENQUIRY'
                    )
                    tmp = tmp.get('action', {})
                    tmp[0]['text'] = (
                        'Well received. '
                        'Your message is viewing by a librarian now, '
                        'will get back to you as soon as possible. '
                        'Thanks for your patience.'
                    )
                    actions.extend(tmp)
                    del tmp
            # 4. if not then return intent response with helpful buttons
            else:
                actions.insert(0, dict(
                    type='text',
                    text=loads(self._model.get_response(intent)).get('text')
                ))
                if intent not in ['smalltalk', 'ask a librarian']:
                    tmp = self._model.get_payload('BUTTON_HELPFUL_CONFIRM')
                    tmp = tmp.get('action', {})
                    actions.extend(tmp)
                    del tmp

        debug(actions, 'actions:')
        fb = FacebookRely(
            self._template, self._results, actions, self._model.memory
        )

        return fb.call()
