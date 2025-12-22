//~ Strings

internal str8 
S8SkipLastSlash(str8 String)
{
	umm LastSlash = 0;
	for EachIndex(i, String.Size)
	{
		if(String.Data[i] == '/')
		{
			LastSlash = i;
		}
	}
	LastSlash += 1;
    
	str8 Result = S8From(String, LastSlash);
	return Result;
}

internal b32
S8Match(str8 A, str8 B, b32 AIsPrefix)
{
    b32 Match = false;
    
    b32 Requirements = false;
    Requirements |= ( AIsPrefix && (A.Size <= B.Size));
    Requirements |= (!AIsPrefix && (A.Size == B.Size));
    
    if(Requirements)
    {
        Match = true;
        for EachIndex(Idx, A.Size)
        {
            if((A.Data[Idx] != B.Data[Idx]))
            {
                Match = false;
                break;
            }
        }
    }
    
    return Match;
}

internal umm
StringLength(char *String)
{
    umm Result = 0;
    
    while(*String)
    {
        String += 1;
        Result += 1;
    }
    
    return Result;
}