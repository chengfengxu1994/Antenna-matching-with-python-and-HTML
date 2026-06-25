"""Merge Optenni into full_components.db"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.murata_db import MurataDatabase

DB = r'E:\RF matching\Murata\full_components.db'
OPTE = r'C:\Users\mocha\AppData\Roaming\Optenni\ComponentLibrary'

db = MurataDatabase(DB)
db.connect()
db._add_manufacturer_column()

c = db.cursor
c.execute("UPDATE series SET manufacturer='Murata' WHERE manufacturer='' OR manufacturer IS NULL")
db.conn.commit()

stats = db.get_statistics()
print("Before merge: %d components" % stats['total_components'])

def pct(cur, tot, msg):
    p = cur/tot*100 if tot>0 else 0
    sys.stdout.write("\r  [%5.1f%%] %s" % (p, msg))
    sys.stdout.flush()
    if cur == tot:
        print()

db.populate_from_optenni_dir(OPTE, progress_callback=pct)

stats = db.get_statistics()
print()
print("After merge: %d components, %d series" % (stats['total_components'], stats['total_series']))
print("Sparam records: %d" % stats['sparam_records'])
print("Primaries: %d" % stats['primary_components'])

# Cleanup
c.execute("DELETE FROM freq_grid WHERE id NOT IN (SELECT MIN(id) FROM freq_grid GROUP BY freq_hz)")
db.conn.commit()

db.close()
print("Done! DB at %s" % DB)
