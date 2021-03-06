#include "grid_structure.h"
#include <cassert>


GridStructure::GridStructure(
        const vector<int>& _horzPositions, 
        const vector<int>& _vertPositions) :
    horzPositions(_horzPositions),
    vertPositions(_vertPositions)
{
    assert(horzPositions.size() >= 2);
    assert(vertPositions.size() >= 2);
    assert(isSorted(horzPositions));
    assert(isSorted(vertPositions));
}

Rect GridStructure::GetCellRect(int rowIndex, int colIndex) const
{
    assert(0 <= rowIndex && rowIndex < GetRowsCount());
    assert(0 <= colIndex && colIndex < GetColsCount());
    return Rect(
        vertPositions[colIndex],
        horzPositions[rowIndex],
        vertPositions[colIndex+1],
        horzPositions[rowIndex+1]
    );
}

bool GridStructure::isSorted(const vector<int>& positions) const
{
    for(int i = 1; i < positions.size(); ++i) {
        if(positions[i-1] > positions[i]) {
            return false;
        }
    }
    return true;
}