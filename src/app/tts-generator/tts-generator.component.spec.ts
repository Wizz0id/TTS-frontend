import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TtsGeneratorComponent } from './tts-generator.component';

describe('TtsGeneratorComponent', () => {
  let component: TtsGeneratorComponent;
  let fixture: ComponentFixture<TtsGeneratorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TtsGeneratorComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TtsGeneratorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
