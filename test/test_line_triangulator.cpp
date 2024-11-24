#include "fstream"
#include "iostream"

#include "line_triangulator.h"
#include "slam_log_reporter.h"

using namespace VISION_GEOMETRY;

int main(int argc, char **argv) {
    ReportInfo(YELLOW ">> Test line triangulator." RESET_COLOR);
    LogFixPercision(3);

    LineTriangulator solver;

    return 0;
}