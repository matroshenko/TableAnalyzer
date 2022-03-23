#pragma once
#include <vector>

using std::vector;


class ConnectedComponentsFinder {
public:
    explicit ConnectedComponentsFinder(const vector<vector<int>>& _graph) : graph(_graph) {}

    vector<vector<int>> Find() const;

private:
    vector<vector<int>> graph;

    vector<int> findConnectedComponent(int source, vector<bool>& visited) const;
};