#import dotenv
import pynamodb.attributes as attrs
import pynamodb.models as models
from dotenv import load_dotenv


class Thingy(models.Model):
    class Meta:
        table_name = 'Thingy'
        host = None

    id = attrs.NumberAttribute(hash_key=True)
    name = attrs.UnicodeAttribute()

    def __repr__(self):
        return "<Thingy {} name:{}>".format(self.id, self.name)


def main():
    #dotenv.load_dotenv()
    load_dotenv()
    if Thingy.exists():
        Thingy.delete_table()
    Thingy.create_table(read_capacity_units=1, write_capacity_units=1, wait=True)

    thang = Thingy(id=0, name="dodo")
    thang.save()

    stored_thingy = Thingy.get(0)
    print(stored_thingy)

    for t in Thingy.scan(Thingy.name == "dodo"):
        print(t)


if __name__ == "__main__":
    main()