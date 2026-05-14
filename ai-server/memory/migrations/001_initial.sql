CREATE TABLE exchanges (
  exchange_id    INTEGER PRIMARY KEY,
  device_id      TEXT NOT NULL,
  ts             INTEGER NOT NULL,
  user_msg       TEXT NOT NULL,
  assistant_msg  TEXT NOT NULL,
  intent         TEXT,
  metadata       TEXT
);

CREATE INDEX idx_exchanges_device_ts ON exchanges(device_id, ts);

CREATE VIRTUAL TABLE exchanges_fts USING fts5(
  user_msg, assistant_msg,
  content='exchanges', content_rowid='exchange_id'
);

CREATE TRIGGER exchanges_ai AFTER INSERT ON exchanges BEGIN
  INSERT INTO exchanges_fts(rowid, user_msg, assistant_msg)
  VALUES (new.exchange_id, new.user_msg, new.assistant_msg);
END;

CREATE TRIGGER exchanges_ad AFTER DELETE ON exchanges BEGIN
  INSERT INTO exchanges_fts(exchanges_fts, rowid, user_msg, assistant_msg)
  VALUES ('delete', old.exchange_id, old.user_msg, old.assistant_msg);
END;

CREATE TRIGGER exchanges_au AFTER UPDATE ON exchanges BEGIN
  INSERT INTO exchanges_fts(exchanges_fts, rowid, user_msg, assistant_msg)
  VALUES ('delete', old.exchange_id, old.user_msg, old.assistant_msg);
  INSERT INTO exchanges_fts(rowid, user_msg, assistant_msg)
  VALUES (new.exchange_id, new.user_msg, new.assistant_msg);
END;
