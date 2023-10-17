import logging
import datetime

my_log = logging.getLogger()
my_log.setLevel(logging.DEBUG)

now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

file_hand = logging.FileHandler(f"./log/{now_str}.log")
file_hand.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
file_hand.setLevel(logging.INFO)

my_log.addHandler(file_hand)
