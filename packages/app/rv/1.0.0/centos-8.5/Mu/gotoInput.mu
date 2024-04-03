use rvui;
use commands;
use extra_commands;

module: gotoInput {

global string _storeAttr = nil;

\: gotoSource (void; string value)
{
    if (_storeAttr neq nil)
    {
        print(_storeAttr);
        setStringProperty(_storeAttr, string[]{value}, true);
    }
}

\: frameInput(void; string attr, Event event)
{
    _storeAttr = attr;
    let enterText = startTextEntryMode(\:(string;) {"Go to GlobalFrame: ";}, gotoSource);
    enterText(event);
}

}   // END MODULE simpleInput
