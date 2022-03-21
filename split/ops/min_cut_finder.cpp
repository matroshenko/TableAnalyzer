#include "min_cut_finder.h"
#include <climits>
#include <algorithm>
#include <cassert>

using std::min;


MinCutFinder::MinCutFinder(const vector<vector<int>>& _graph, 
        const unordered_map<pair<int, int>, int>& _capacity):
    graph(_graph),
    capacity(_capacity)
{
}

vector<bool> MinCutFinder::Find(int s, int t) const
{
    const int numOfNodes = graph.size();

    unordered_map<pair<int, int>, int> flow(capacity);
    for(auto& item : flow) {
        item.second = 0;
    }

    vector<int> parent(numOfNodes);
    while(bfs(s, t, flow, parent)) {
        // Find the maximum flow through the path found.
        int pathFlow = INT_MAX;
        for(int v = t; v != s; v = parent[v]) {
            const int u = parent[v];
            pathFlow = min(pathFlow, capacity.at({u, v}) - flow.at({u, v}));
        }
        assert(pathFlow > 0);

        // update flow of the edges and reverse edges
        // along the path
        for(int v = t; v != s; v = parent[v]) {
            const int u = parent[v];
            flow.at({u, v}) += pathFlow;
            flow.at({v, u}) -= pathFlow;
        }
    }
    // Flow is maximum now.
    // Find vertices reachable from s.
    vector<bool> result(numOfNodes, false);
    dfs(s, flow, result);
        
    return result;
}