from json import JSONEncoder

from core.game import GameHistory


class GameHistoryEncoder(JSONEncoder):
    def default(self, game_history: GameHistory):
        return {
            'num_actions': game_history.actions.__len__()
        }
