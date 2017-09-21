#!/usr/local/bin/python3
# Payload Handler of the Chatbot Webhook APP
# Created By Intralogue

from chatbot.helper import debug

from .facebook import FacebookRely
from .search import Search
from . import original


class Payload:
    def __init__(self, original, payload, model, template, results):
        self._actions, self._payload = model.get_payload(payload).values()
        self._model = model
        self._template = template
        self._results = results
        self._original = original
        self.msg = original
        self.predictions = []
        self.general_slots = []
        if self._payload:
            self.msg = ''
        debug('Incoming payload: {}'.format(self._payload))

    def process(self):
        memory = self._model.memory
        if not self._payload:
            fb = FacebookRely(
                self._template,
                self._results,
                self._actions,
                memory
            )

            return fb.call()

        if self._payload.startswith('BUTTON_SEARCH_REFINE_'):
            lastaction = self._actions[-1]
            if ('postfix' in lastaction and
                    lastaction.get('type') == 'quick_reply'):
                step = lastaction.get('postfix')
                for item in memory[step[0]][step[1]]:
                    lastaction['buttons'].append([
                        item, 'BUTTON_{}_REFINEMENT_{}'.format(
                            self._payload.split('_')[-1], item
                        )
                    ])

        if 'REFINEMENT_' in self._payload:
            search_result = self._model.get_payload(
                'SEARCH_RESULT'
            ).get('action')
            if self._actions and 'context' in self._actions[0]:
                memory['context'] = self._actions[0]['context']
            # 1. update keyword with new refinement
            keyword = self._payload.split('_')[-1]
            (
                __,
                book_keyword,
                self.predictions,
                self.general_slots,
                __
            ) = original.parser(keyword)

            memory['cquery']['refinements'].append(book_keyword)
            memory['cquery']['keyword'] = book_keyword
            # 2. add the search result in between
            search = Search(
                self._original,
                memory.get('context'),
                memory.get('cquery')
            )
            authors, date, elems, stype = search.start(refine=True)
            # 3. update the text if the search result is empty
            if elems:
                search_result[0]['text'] = search_result[0]['text'].format(
                    num=len(elems)
                )
                search_result.insert(1, {
                    'type': 'search',
                    'elems': elems,
                    'context': 'Refine Search ' + stype
                })
            else:
                search_result[0]['text'] = search_result[0]['error']

            self._actions = search_result

        fb = FacebookRely(
            self._template,
            self._results,
            self._actions,
            memory
        )

        return fb.call()
