#include "rect.h"
#include <cassert>

Rect::Rect(int left, int top, int right, int bottom) :
    Left(left),
    Top(top),
    Right(right),
    Bottom(bottom)
{
    assert(Left <= Right && Top <= Bottom);
}

bool Rect::IsEmpty() const
{
    return Left == Right || Top == Bottom;
}