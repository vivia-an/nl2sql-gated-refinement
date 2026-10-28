from nl2sql.analysis.linking import linking_recall

SCHEMA = {"Customers": [{"name": "id"}, {"name": "name"}], "Orders": [{"name": "id"}, {"name": "cust_id"}, {"name": "total"}],
          "Items": [{"name": "order_id"}, {"name": "sku"}]}
SQL = "SELECT T1.name, SUM(T2.total) FROM Customers AS T1 JOIN Orders AS T2 ON T1.id = T2.cust_id GROUP BY T1.name"


def test_full_and_partial_context():
    assert linking_recall(SQL, SCHEMA, None, True) == {"table_recall": 1.0, "column_recall": 1.0}
    r = linking_recall(SQL, SCHEMA, ["Orders", "Items"], False)
    assert r["table_recall"] == 0.5 and 0 < r["column_recall"] < 1
    assert linking_recall(SQL, SCHEMA, ["customers", "orders"], False) == {"table_recall": 1.0, "column_recall": 1.0}


def test_unparseable_gold_is_none():
    assert linking_recall("SELEKT ((", SCHEMA, ["Orders"], False) is None
