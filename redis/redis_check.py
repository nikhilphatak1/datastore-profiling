import redis
import pdb

if __name__ == "__main__":
    #pdb.set_trace()
    r = redis.Redis(host="127.0.0.1",port=6379)
    r.set("boo","mark")
    print(r.get("boo"))
    print("Finish Line")

