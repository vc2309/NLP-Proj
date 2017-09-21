#!/usr/local/bin/python3
# Message Processer of the Chatbot Webhook APP
# Created By Intralogue

# from json import loads

from chatbot.helper import NONE

from .payload import Payload
from .context import Context


class Message:
    def __init__(self, evt, model):
        msg = evt.get('message', {})
        self.original = msg.get('text')
        self.results = []
        self._payload = msg.get(
            'quick_reply', evt.get('postback', {})
        ).get('payload')

        self.template = {
            'message': dict(
                text=None,
                quick_replies=[]
            ),
            'context': NONE
        }

        self.payload = Payload(
            original=self.original,
            payload=self._payload,
            results=self.results,
            template=self.template,
            model=model
        )

        self.context = Context(
            results=self.results,
            template=self.template,
            model=model,
            originalmsg=self.payload.msg
        )

        self.cache = dict(original_message=self.original)

    def analyse(self):
        message = self._payload
        # 1. check for payload
        context = self.payload.process()
        predictions = self.payload.predictions
        slots = self.payload.general_slots

        # 2. check for context
        if not self.results:
            context = self.context.process()
            message = self.context.message
            predictions = self.context.class_probability
            slots = self.context.general_slots

        # 3. get back payload and context result
        if not self.results:
            result = dict(message={})
            context = 'Ask Librarian'
            result['message']['text'] = (
                'Regarding to your information given, '
                'we may direct you to the librarian for assistance. '
                'Please leave your message here. Thank you.'
            )
            self.results.append(result)

        self.cache = dict(
            message=message,
            original_message=self.original,
            slots=slots,
            predictions=predictions
        )

        return context, self.results, self.cache
