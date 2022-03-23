#pragma once

#include "tensorflow/core/framework/op.h"
#include "rect.h"
#include <vector>
#include <utility>

using namespace tensorflow;
using std::vector;
using std::pair;

class CellsStructureBuilder {
public:
    CellsStructureBuilder(const Tensor& _mergeRightMask, const Tensor& _mergeDownMask);

    vector<Rect> Build() const;

private:
    const Tensor& mergeRightMask;
    const Tensor& mergeDownMask;
    int rowsCount;
    int colsCount;

    vector<Rect> buildInitialCells() const;

    vector<vector<int>> createGraph() const;
    pair<int, int> to2DIndex(int index) const;
    int to1DIndex(int rowIndex, int colIndex) const;

    Rect getComponentRect(const vector<int>& component) const;
    
    void mergeIntersectingCells(vector<Rect>& cells) const;
    bool findIntersectingCells(const vector<Rect>& cells, int& firstIndex, int& secondIndex) const;
};