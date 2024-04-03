//----------//
// ZDebugLog.h //
//-------------------------------------------------------//
// author: Wanho Choi   @ Dexter Studios                 //
//         Dohyeon Yang @ Dexter Studios                 //
// last update: 2017.04.06                               //
//-------------------------------------------------------//
#ifndef _ZDebugLOG_h_
#define _ZDebugLOG_h_
#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN
class ZDebugLog
{
public:
    bool          mOn;

    std::stack<std::string>	mWork_stack;
    std::stack<clock_t>	mTime_stack;

    clock_t	mTime;
    int mTab_size;
    int mSpace_level;
    std::string mPrev_work;

public:
    ZDebugLog(bool on = true);
    ~ZDebugLog();
    void print_tab(int tab_size);
    void in(const char *fmt, ...);
    void out(void);
    void message(const char *fmt, ...);
    std::string format_arg_list(const char *fmt, va_list args);
};

ZELOS_NAMESPACE_END
#endif

