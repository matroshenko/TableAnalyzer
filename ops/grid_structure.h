#pragma once

#include "rect.h"
#include <vector>

using std::vector;

class GridStructure {
public:
    GridStructure(const vector<int>& _horzPositions, const vector<int>& _vertPositions);

    int GetRowsCount() const { return horzPositions.size() - 1; }
    int GetColsCount() const { return vertPositions.size() - 1; }

    Rect GetCellRect(int rowIndex, int colIndex) const;

private:
    vector<int> horzPositions;
    vector<int> vertPositions;

    bool isSorted(const vector<int>& positions) const;
};