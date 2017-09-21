#!/usr/local/bin/python3
# Facebook webhook response handling
# Created by Intralogue

from urllib.parse import urlparse as purl
import math
import time

import requests

from chatbot.helper import debug


class FacebookRely:
    def __init__(self, template, results, actions, memory):
        que = []
        self.que = que
        self.template = template.copy()
        self.results = results
        self.reply = None
        self.context = None
        self.memory = memory
        for action in actions:
            self.reply = action.get('text')
            self.context = action.get('context', self.context)
            caller = action.get('type', 'error')
            if hasattr(self, caller):
                que.append([getattr(self, caller), action])
            else:
                que.append([getattr(self, 'error'), caller])

    def call(self):
        for que in self.que:
            self.datetime = str(math.floor(time.time() * 1000))
            result = que[0](que[1])
            if result:
                del result['context']
                result['datetime'] = self.datetime
                self.results.append(result)

        return self.context

    def postback(self, action):
        debug('Creating postback send API object')
        if 'buttons' in action:
            result = self.template.copy()
            result['message']['attachment'] = dict(type='template', payload={})
            result['message']['attachment']['payload'] = dict(
                text=action.get('text'),
                template_type='button',
                buttons=[]
            )
            for button in action.get('buttons'):
                if 'postfix' in action:
                    button[1] += self.datetime
                result['message']['attachment']['payload']['buttons'].append({
                    'type': 'postback',
                    'title': button[0],
                    'payload': button[1]
                })
            return result

    def quick_reply(self, action):
        debug('Creating quick_reply send API object')
        if 'buttons' in action:
            if 'text' in action:
                result = self.template.copy()
                result['message']['text'] = action['text']
            else:
                result = self.results[-1]
            for button in action.get('buttons'):
                result['message']['quick_replies'].append({
                    'content_type': 'text',
                    'title': button[0],
                    'payload': button[1]
                })

            if 'text' in action:
                return result

    def search(self, action):
        debug('Creating search send API object')
        result = {
            'message': {
                'attachment': {
                    'type': 'template',
                    'payload': {
                        'template_type': 'generic',
                        'image_aspect_ratio': 'square',
                        'elements': []
                    }
                },
                'quick_replies': []
            },
            'context': action.get('context')
        }

        for elem in action.get('elems'):
            result['message']['attachment']['payload']['elements'].append({
                'title': elem.get('title', 'LEXICA'),
                'image_url': elem.get('image', action.get('image')),
                'subtitle': elem.get('subtitle', 'LEXICA'),
                'buttons': [{
                    'type': 'web_url',
                    'url': elem.get('url'),
                    'title': 'View'
                }, {
                    'type': 'postback',
                    'title': 'Get a Summary',
                    'payload': 'SUMMARY_' + elem.get('uuid')
                }, {
                    'type': 'postback',
                    'title': 'Reserve a Copy',
                    'payload': 'RESERVE_COPY'
                }]
            })

        return result

    def text(self, action):
        debug('Creating text send API object')
        if 'text' in action:
            result = self.template.copy()
            result['message']['text'] = action.get('text')
            return result

    def image(self, action):
        debug('Creating image send API object')
        if 'text' in action:
            scheme, *__ = purl(action.get('text'))
            result = self.template.copy()
            result['message']['attachment'] = {
                'type': 'image',
                'payload': (
                    {'url': action.get('text')}
                    if scheme in ['http', 'https'] else
                    {'attachment_id': action.get('text')}
                )
            }
            return result

    def error(self, caller):
        raise ValueError(
            (
                'Caller({}) '
                'not match any of the current existing.'
            ).format(caller)
        )


def get_user_details(token, sender_id):
    r = requests.get(
        'https://graph.facebook.com/v2.6/' + str(sender_id),
        params={"access_token": token}
    )
    return r.json()
