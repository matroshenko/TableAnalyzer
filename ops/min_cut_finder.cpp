#include "min_cut_finder.h"
#include <climits>
#include <algorithm>
#include <cassert>
#include <queue>

using std::min;
using std::queue;


MinCutFinder::MinCutFinder(const vector<vector<int>>& _graph, 
        const TCapacity& _capacity):
    graph(_graph),
    capacity(_capacity)
{
}

vector<bool> MinCutFinder::Find(int s, int t) const
{
    const int numOfNodes = graph.size();

    TCapacity flow(capacity);
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

bool MinCutFinder::bfs(
    int s, int t, const TCapacity& flow, 
    vector<int>& parent) const
{
    const int numOfNodes = graph.size();
    vector<bool> visited(numOfNodes, false);

    queue<int> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;

    while(q.size() > 0) {
        const int u = q.front();
        if(u == t) {
            return true;
        }
        q.pop();

        for(int v : graph[u]) {
            if(!visited[v] && flow.at({u, v}) < capacity.at({u, v})) {
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
   }
   return false;
}

void MinCutFinder::dfs(
    int u, const TCapacity& flow, vector<bool>& visited) const
{
    visited[u] = true;
    for(int v : graph[u]) {
        if(!visited[v] && flow.at({u, v}) < capacity.at({u, v})) {
            dfs(v, flow, visited);
        }
    }
}