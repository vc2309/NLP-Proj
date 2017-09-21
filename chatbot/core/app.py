#!/usr/local/bin/python3
# Core Function of the Chatbot Webhook APP
# Created By Intralogue

import requests
import json
import time

from chatbot.helper import debug, get_setup, clean_empty

from .facebook import get_user_details

from .model import Model
from .message import Message


class ChatBot:
    def __init__(self, sid, rid):
        self._sid = sid
        self._rid = rid
        (
            self._ACCESS_TOKEN,
            self._UNIVERSITY,
            self._Special_case_list
        ) = get_setup(rid, sid)

        if not self._ACCESS_TOKEN:
            debug('Recipient ({}) does not exist.'.format(rid), 'WARNING')
            return ('Done', 200)

        self._model = Model(rid, sid, self._UNIVERSITY)
        self._sender_details = get_user_details(self._ACCESS_TOKEN, sid)

    def receive(self, evt):
        message = Message(evt, self._model)
        # get back results
        context, results, cache = message.analyse()

        # save last result to dynamodb
        cache['sender'] = self._sid
        cache['datetime'] = int(results[-1].get('datetime', time.time()))
        cache['reply'] = results[-1]
        self.cache = cache.copy()
        del cache

        # for each result send api request
        for result in results:
            self.send(result)

        # print out the context and update
        memory = self._model.memory
        memory['context'] = context or memory['context']
        del context
        debug(memory.get('context'), 'context')

        # update dynamodb memory
        self._model.upload_memory()
        # return 200
        debug(
            evt, 'Finished process request from sender({}):'.format(self._sid)
        )
        return ('Offical Response.', 200)

    def send(self, msg):
        msg['recipient'] = {
            'id': self._sid
        }
        msg = clean_empty(msg)
        debug(msg, 'Before send to Send API')
        res = requests.post(
            'https://graph.facebook.com/v2.6/me/messages',
            params={"access_token": self._ACCESS_TOKEN},
            data=json.dumps(msg),
            headers={'Content-type': 'application/json'}
        )

        if res.status_code != requests.codes.ok:
            res = res.json()
            raise ValueError(
                'Send API response error: {}'.format(res['error']['message'])
            )
        else:
            self._model.create_cache(self.cache)
