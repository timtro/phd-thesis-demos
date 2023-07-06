#! /bin/env bash

tail -n1 pid-test-a.csv pid-test-b.csv pid-test-c.csv pid-test-d.csv | awk -F ', ' '/^[0-9]+, [0-9]+\.[0-9]+, [0-9]+\.[0-9]+$/ { error = ($2 - $3) / $3 * 100; printf("%.3f%%\n", error); next } { print }'
